package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class ProductInfo_printProductList_10_1_Test {

    @InjectMocks
    private ProductInfo productInfo;

    @Mock
    private ArrayList<ProductDetails> products;

    @Mock
    private ProductDetails productDetails;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        productInfo.setTotalResults("10");
        productInfo.setTotalPages("2");
    }

    @Test
    void testPrintProductListWithProducts() {
        when(products.size()).thenReturn(2);
        when(products.get(0)).thenReturn(productDetails);
        when(products.get(1)).thenReturn(productDetails);
        when(productDetails.getAsin()).thenReturn("123");
        when(productDetails.getProductName()).thenReturn("Product 1");
        when(productDetails.getOurPrice()).thenReturn("10.00");
        productInfo.setDetails(new ProductDetails[] { productDetails, productDetails });
        String expectedOutput = "Total results = 10\n" + "Total pages = 2\n" + "< 0 > 123 : Product 1 - 10.00\n" + "< 1 > 123 : Product 1 - 10.00\n" + "# of products = 2\n";
        String actualOutput = productInfo.printProductList();
        assertEquals(expectedOutput, actualOutput);
    }
}
