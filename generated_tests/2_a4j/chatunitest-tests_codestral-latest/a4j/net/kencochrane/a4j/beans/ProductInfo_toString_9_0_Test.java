package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class ProductInfo_toString_9_0_Test {

    @InjectMocks
    private ProductInfo productInfo;

    @Mock
    private ArrayList<ProductDetails> products;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        productInfo.setTotalResults("10");
        productInfo.setTotalPages("2");
    }

    @Test
    void testToStringWithProducts() {
        ProductDetails product1 = mock(ProductDetails.class);
        ProductDetails product2 = mock(ProductDetails.class);
        when(products.size()).thenReturn(2);
        when(products.get(0)).thenReturn(product1);
        when(products.get(1)).thenReturn(product2);
        when(product1.toString()).thenReturn("Product 1 Details");
        when(product2.toString()).thenReturn("Product 2 Details");
        productInfo.setDetails(new ProductDetails[] { product1, product2 });
        String expected = "Total results = 10\n" + "Total pages = 2\n" + "Product 1 Details\n" + "Product 2 Details\n" + "# of products = 2\n";
        assertEquals(expected, productInfo.toString());
    }
}
