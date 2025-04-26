package net.kencochrane.a4j;

import net.kencochrane.a4j.DAO.Search;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.beans.*;

@ExtendWith(MockitoExtension.class)
class A4j_BlendedSearch_1_0_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    @Test
    void blendedSearch_withValidInput_returnsBlendedSearchResult() {
        String searchTerm = "testTerm";
        String type = "testType";
        BlendedSearch expectedResult = new BlendedSearch();
        when(search.Blended(searchTerm, type)).thenReturn(expectedResult);
        BlendedSearch result = a4j.BlendedSearch(searchTerm, type);
        assertEquals(expectedResult, result);
    }

    @Test
    void blendedSearch_withNullSearchTerm_returnsBlendedSearchResult() {
        String searchTerm = null;
        String type = "testType";
        BlendedSearch expectedResult = new BlendedSearch();
        when(search.Blended(searchTerm, type)).thenReturn(expectedResult);
        BlendedSearch result = a4j.BlendedSearch(searchTerm, type);
        assertEquals(expectedResult, result);
    }

    @Test
    void blendedSearch_withNullType_returnsBlendedSearchResult() {
        String searchTerm = "testTerm";
        String type = null;
        BlendedSearch expectedResult = new BlendedSearch();
        when(search.Blended(searchTerm, type)).thenReturn(expectedResult);
        BlendedSearch result = a4j.BlendedSearch(searchTerm, type);
        assertEquals(expectedResult, result);
    }

    @Test
    void blendedSearch_withNullSearchTermAndType_returnsBlendedSearchResult() {
        String searchTerm = null;
        String type = null;
        BlendedSearch expectedResult = new BlendedSearch();
        when(search.Blended(searchTerm, type)).thenReturn(expectedResult);
        BlendedSearch result = a4j.BlendedSearch(searchTerm, type);
        assertEquals(expectedResult, result);
    }

    @Test
    void blendedSearch_withEmptySearchTerm_returnsBlendedSearchResult() {
        String searchTerm = "";
        String type = "testType";
        BlendedSearch expectedResult = new BlendedSearch();
        when(search.Blended(searchTerm, type)).thenReturn(expectedResult);
        BlendedSearch result = a4j.BlendedSearch(searchTerm, type);
        assertEquals(expectedResult, result);
    }

    @Test
    void blendedSearch_withEmptyType_returnsBlendedSearchResult() {
        String searchTerm = "testTerm";
        String type = "";
        BlendedSearch expectedResult = new BlendedSearch();
        when(search.Blended(searchTerm, type)).thenReturn(expectedResult);
        BlendedSearch result = a4j.BlendedSearch(searchTerm, type);
        assertEquals(expectedResult, result);
    }

    @Test
    void blendedSearch_withEmptySearchTermAndType_returnsBlendedSearchResult() {
        String searchTerm = "";
        String type = "";
        BlendedSearch expectedResult = new BlendedSearch();
        when(search.Blended(searchTerm, type)).thenReturn(expectedResult);
        BlendedSearch result = a4j.BlendedSearch(searchTerm, type);
        assertEquals(expectedResult, result);
    }
}
