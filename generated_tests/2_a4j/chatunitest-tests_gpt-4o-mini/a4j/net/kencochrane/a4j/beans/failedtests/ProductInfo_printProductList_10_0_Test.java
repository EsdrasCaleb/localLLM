package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class ProductInfo_printProductList_10_0_Test {

    private ProductInfo productInfo;

    @BeforeEach
    void setUp() {
        productInfo = new ProductInfo();
    }

    @Test
    void testPrintProductList_withProducts() {
        // Arrange
        ProductDetails product1 = Mockito.mock(ProductDetails.class);
        Mockito.when(product1.getAsin()).thenReturn("B001");
        Mockito.when(product1.getProductName()).thenReturn("Product 1");
        Mockito.when(product1.getOurPrice()).thenReturn("$10.00");
        ProductDetails product2 = Mockito.mock(ProductDetails.class);
        Mockito.when(product2.getAsin()).thenReturn("B002");
        Mockito.when(product2.getProductName()).thenReturn("Product 2");
        Mockito.when(product2.getOurPrice()).thenReturn("$20.00");
        productInfo.setTotalResults("2");
        productInfo.setTotalPages("1");
        productInfo.setDetails(new ProductDetails[] { product1, product2 });
        // Act
        String result = productInfo.printProductList();
        // Assert
        String expected = "Total results = 2\n" + "Total pages = 1\n" + "< 0 > B001 : Product 1 - $10.00\n" + "< 1 > B002 : Product 2 - $20.00\n" + "# of products = 2\n";
        assertEquals(expected, result);
    }

    @Test
    void testPrintProductList_noProducts() {
        // Arrange
        productInfo.setTotalResults("0");
        productInfo.setTotalPages("0");
        productInfo.setDetails(new ProductDetails[] {});
        // Act
        String result = productInfo.printProductList();
        // Assert
        String expected = "Total results = 0\n" + "Total pages = 0\n" + "# of products = 0\n";
        assertEquals(expected, result);
    }

    @Test
    void testPrintProductList_productsNull() {
        // Arrange
        productInfo.setTotalResults("0");
        productInfo.setTotalPages("0");
        productInfo.setDetails(null);
        // Act
        String result = productInfo.printProductList();
        // Assert
        String expected = "Total results = 0\n" + "Total pages = 0\n" + "products is null \n";
        assertEquals(expected, result);
    }
}
