package net.kencochrane.a4j;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

@ExtendWith(MockitoExtension.class)
public class A4j_BlendedSearch_1_4_Test {

    @Mock
    Search search;

    @InjectMocks
    A4j a4j;

    @Test
    public void testBlendedSearch() {
        // Given
        String searchTerm = "test";
        String type = "testType";
        BlendedSearch expectedBlendedSearch = new BlendedSearch();
        when(search.Blended(searchTerm, type)).thenReturn(expectedBlendedSearch);
        // When
        BlendedSearch actualBlendedSearch = a4j.BlendedSearch(searchTerm, type);
        // Then
        assertEquals(expectedBlendedSearch, actualBlendedSearch);
    }
}
