package net.kencochrane.a4j;

import java.util.ArrayList;
import java.util.List;
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
class A4j_AuthorSearch_5_2_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    @Test
    void testAuthorSearch_validInput_returnsProductInfo() {
        ProductInfo productInfo = new ProductInfo();
        // Adding a list name for completeness
        productInfo.setListName("testList");
        List<ProductDetails> productList = new ArrayList<>();
        // Adding a sample ProductDetails object
        productList.add(new ProductDetails());
        productList.add(new ProductDetails());
        productInfo.setDetails(productList.toArray(new ProductDetails[0]));
        productInfo.setTotalResults("10");
        productInfo.setTotalPages("2");
        Mockito.when(search.AuthorSearch("Test Author", "1")).thenReturn(productInfo);
        ProductInfo result = a4j.AuthorSearch("Test Author", "1");
        assertEquals(productInfo, result);
    }

    @Test
    void testAuthorSearch_emptyAuthorName_returnsNull() {
        Mockito.when(search.AuthorSearch("", "1")).thenReturn(null);
        ProductInfo result = a4j.AuthorSearch("", "1");
        assertNull(result);
    }

    @Test
    void testAuthorSearch_invalidPage_returnsNull() {
        Mockito.when(search.AuthorSearch("Test Author", "abc")).thenReturn(null);
        ProductInfo result = a4j.AuthorSearch("Test Author", "abc");
        assertNull(result);
    }
}
