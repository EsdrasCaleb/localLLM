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
public class A4j_KeywordSearch_2_0_Test {

    @Mock
    Search search;

    @InjectMocks
    A4j a4j;

    @Test
    public void testKeywordSearch() {
        String searchTerm = "test";
        String productLine = "test";
        String type = "test";
        String page = "test";
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.Keyword(searchTerm, productLine, type, page)).thenReturn(expectedProductInfo);
        ProductInfo actualProductInfo = a4j.KeywordSearch(searchTerm, productLine, type, page);
        assertEquals(expectedProductInfo, actualProductInfo);
    }
}
