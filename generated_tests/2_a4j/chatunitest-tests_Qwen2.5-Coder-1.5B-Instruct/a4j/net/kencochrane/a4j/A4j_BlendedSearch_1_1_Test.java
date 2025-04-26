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

class A4j_BlendedSearch_1_1_Test {

    private A4j a4j;

    private Search mockSearch;

    @BeforeEach
    public void setUp() {
        a4j = new A4j();
        mockSearch = mock(Search.class);
        when(mockSearch.Blended(anyString(), anyString())).thenReturn(new BlendedSearch());
    }

    @Test
    public void testBlendedSearch() {
        BlendedSearch blendedSearch = a4j.BlendedSearch("testTerm", "testType");
        assertNotNull(blendedSearch);
    }
}
