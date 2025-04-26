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

class A4j_ListmaniaSearch_9_0_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testListmaniaSearch() {
        String listId = "testListId";
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.ListmaniaSearch(listId)).thenReturn(expectedProductInfo);
        ProductInfo result = a4j.ListmaniaSearch(listId);
        assertEquals(expectedProductInfo, result);
        verify(search, times(1)).ListmaniaSearch(listId);
    }
}
