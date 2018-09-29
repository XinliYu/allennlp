"""
An ``AtisSqlTableContext`` represents the SQL context in which an utterance appears
for the Atis dataset, with the grammar and the valid actions.
"""
from typing import List, Dict
import sqlite3
from copy import deepcopy

from overrides import overrides
from parsimonious.grammar import Grammar

from allennlp.common.file_utils import cached_path
from allennlp.semparse.contexts.sql_context_utils import initialize_valid_actions, format_grammar_string
from allennlp.semparse.contexts.sql_context_utils import SqlTableContext

# This is the base definition of the SQL grammar in a simplified sort of
# EBNF notation, and represented as a dictionary. The keys are the nonterminals and the values
# are the possible expansions of the nonterminal where each element in the list is one possible expansion.
# Rules that differ only in capitalization of keywords are mapped to the same action by
# the ``SqlVisitor``.  The nonterminal of the first rule is the starting symbol.
# In addition to the grammar here, we add ``col_ref``, ``table_name`` based on the tables
# that ``SqlTableContext`` is initialized with. ``number`` is initialized to
# be empty and later on updated based on the utterances. ``biexpr`` is altered based on the
# database to column references with strings that are allowed to appear in that column.
# We then create additional nonterminals for each column that may be used as a string constraint
# in the query.
# For example, to include city names as strings:
#
#       grammar_dictionary['biexpr'] = \
#               ['( "city" ws "." ws "city_name"  binop ws city_city_name_strings )',  ...
#       grammar_dictionary['city_city_name_strings'] = ['"NASHVILLE"', '"BOSTON"',  ...

GRAMMAR_DICTIONARY = {}
GRAMMAR_DICTIONARY['statement'] = ['query ws ";" ws']
GRAMMAR_DICTIONARY['query'] = ['(ws "(" ws "SELECT" ws distinct ws select_results ws '
                               '"FROM" ws table_refs ws where_clause ws ")" ws)',
                               '(ws "SELECT" ws distinct ws select_results ws '
                               '"FROM" ws table_refs ws where_clause ws)']
GRAMMAR_DICTIONARY['select_results'] = ['col_refs', 'agg']
GRAMMAR_DICTIONARY['agg'] = ['agg_func ws "(" ws col_ref ws ")"']
GRAMMAR_DICTIONARY['agg_func'] = ['"MIN"', '"min"', '"MAX"', '"max"', '"COUNT"', '"count"']
GRAMMAR_DICTIONARY['col_refs'] = ['(col_ref ws "," ws col_refs)', '(col_ref)']
GRAMMAR_DICTIONARY['table_refs'] = ['(table_name ws "," ws table_refs)', '(table_name)']
GRAMMAR_DICTIONARY['where_clause'] = ['("WHERE" ws "(" ws conditions ws ")" ws)', '("WHERE" ws conditions ws)']
GRAMMAR_DICTIONARY['conditions'] = ['(condition ws conj ws conditions)',
                                    '(condition ws conj ws "(" ws conditions ws ")")',
                                    '("(" ws conditions ws ")" ws conj ws conditions)',
                                    '("(" ws conditions ws ")")',
                                    '("not" ws conditions ws )',
                                    '("NOT" ws conditions ws )',
                                    'condition']
GRAMMAR_DICTIONARY['condition'] = ['in_clause', 'ternaryexpr', 'biexpr']
GRAMMAR_DICTIONARY['in_clause'] = ['(ws col_ref ws "IN" ws query ws)']
GRAMMAR_DICTIONARY['biexpr'] = ['( col_ref ws binaryop ws value)', '(value ws binaryop ws value)']
GRAMMAR_DICTIONARY['binaryop'] = ['"+"', '"-"', '"*"', '"/"', '"="',
                                  '">="', '"<="', '">"', '"<"', '"is"', '"IS"']
GRAMMAR_DICTIONARY['ternaryexpr'] = ['(col_ref ws "not" ws "BETWEEN" ws value ws "AND" ws value ws)',
                                     '(col_ref ws "NOT" ws "BETWEEN" ws value ws "AND" ws value ws)',
                                     '(col_ref ws "BETWEEN" ws value ws "AND" ws value ws)']
GRAMMAR_DICTIONARY['value'] = ['("not" ws pos_value)', '("NOT" ws pos_value)', '(pos_value)']
GRAMMAR_DICTIONARY['pos_value'] = ['("ALL" ws query)', '("ANY" ws query)', 'number',
                                   'boolean', 'col_ref', 'agg_results', '"NULL"']
GRAMMAR_DICTIONARY['agg_results'] = ['(ws "("  ws "SELECT" ws distinct ws agg ws '
                                     '"FROM" ws table_name ws where_clause ws ")" ws)',
                                     '(ws "SELECT" ws distinct ws agg ws "FROM" ws table_name ws where_clause ws)']
GRAMMAR_DICTIONARY['boolean'] = ['"true"', '"false"']
GRAMMAR_DICTIONARY['ws'] = [r'~"\s*"i']
GRAMMAR_DICTIONARY['conj'] = ['"AND"', '"OR"']
GRAMMAR_DICTIONARY['distinct'] = ['("DISTINCT")', '("")']
GRAMMAR_DICTIONARY['number'] = ['""']

KEYWORDS = ['"SELECT"', '"FROM"', '"MIN"', '"MAX"', '"COUNT"', '"WHERE"', '"NOT"', '"IN"', '"LIKE"',
            '"IS"', '"BETWEEN"', '"AND"', '"ALL"', '"ANY"', '"NULL"', '"OR"', '"DISTINCT"']

@SqlTableContext.register("atis")
class AtisSqlTableContext(SqlTableContext):
    """
    An ``AtisSqlTableContext`` represents the SQL context with a grammar of SQL and the valid actions
    based on the schema of the tables that it represents.

    Parameters
    ----------
    all_tables: ``Dict[str, List[str]]``
        A dictionary representing the SQL tables in the dataset, the keys are the names of the tables
        that map to lists of the table's column names.
    tables_with_strings: ``Dict[str, List[str]]``
        A dictionary representing the SQL tables that we want to generate strings for. The keys are the
        names of the tables that map to lists of the table's column names.
    database_file : ``str``, optional
        The directory to find the sqlite database file. We query the sqlite database to find the strings
        that are allowed.
    """
    def __init__(self,
                 all_tables: Dict[str, List[str]] = None,
                 tables_with_strings: Dict[str, List[str]] = None,
                 database_file: str = None) -> None:
        self.grammar_dictionary = deepcopy(GRAMMAR_DICTIONARY)
        self.all_tables = all_tables
        self.tables_with_strings = tables_with_strings
        if database_file:
            self.database_file = cached_path(database_file)
            self.connection = sqlite3.connect(self.database_file)
            self.cursor = self.connection.cursor()

        self.grammar_str: str = self.initialize_grammar_str()
        self.grammar: Grammar = Grammar(self.grammar_str)
        self.valid_actions: Dict[str, List[str]] = initialize_valid_actions(self.grammar, KEYWORDS)
        if database_file:
            self.connection.close()

    @overrides
    def get_grammar_dictionary(self) -> Dict[str, List[str]]:
        return self.grammar_dictionary

    @overrides
    def get_valid_actions(self) -> Dict[str, List[str]]:
        return self.valid_actions

    def initialize_grammar_str(self) -> str:
        if self.all_tables:
            self.grammar_dictionary['table_name'] = \
                    sorted([f'"{table}"'
                            for table in list(self.all_tables.keys())], reverse=True)
            self.grammar_dictionary['col_ref'] = ['"*"']
            for table, columns in self.all_tables.items():
                self.grammar_dictionary['col_ref'].extend([f'("{table}" ws "." ws "{column}")'
                                                           for column in columns])
            self.grammar_dictionary['col_ref'] = sorted(self.grammar_dictionary['col_ref'], reverse=True)

        biexprs = []
        if self.tables_with_strings:
            for table, columns in self.tables_with_strings.items():
                biexprs.extend([f'("{table}" ws "." ws "{column}" ws binaryop ws {table}_{column}_string)'
                                for column in columns])
                for column in columns:
                    self.cursor.execute(f'SELECT DISTINCT {table} . {column} FROM {table}')
                    if column.endswith('number'):
                        self.grammar_dictionary[f'{table}_{column}_string'] = \
                                sorted([f'"{str(row[0])}"' for row in self.cursor.fetchall()], reverse=True)
                    else:
                        self.grammar_dictionary[f'{table}_{column}_string'] = \
                                sorted([f'"\'{str(row[0])}\'"' for row in self.cursor.fetchall()], reverse=True)

        self.grammar_dictionary['biexpr'] = sorted(biexprs, reverse=True) + \
                ['( col_ref ws binaryop ws value)', '(value ws binaryop ws value)']
        return format_grammar_string(self.grammar_dictionary)