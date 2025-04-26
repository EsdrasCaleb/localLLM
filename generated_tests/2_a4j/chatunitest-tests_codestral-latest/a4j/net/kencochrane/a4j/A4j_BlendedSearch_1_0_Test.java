package net.kencochrane.a4j;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

class A4j_BlendedSearch_1_0_Test {

    private A4j a4j;

    private Search searchMock;

    @BeforeEach
    void setUp() {
        a4j = new A4j();
        searchMock = mock(Search.class);
        // Use reflection to set the private search field in A4j
        try {
            java.lang.reflect.Field searchField = A4j.class.getDeclaredField("search");
            searchField.setAccessible(true);
            searchField.set(a4j, searchMock);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
    }

    @Test
    void testBlendedSearch() {
        String searchTerm = "example";
        String type = "type1";
        BlendedSearch expectedBlendedSearch = new BlendedSearch();
        when(searchMock.Blended(searchTerm, type)).thenReturn(expectedBlendedSearch);
        BlendedSearch result = a4j.BlendedSearch(searchTerm, type);
        assertEquals(expectedBlendedSearch, result);
        verify(searchMock, times(1)).Blended(searchTerm, type);
    }
}
