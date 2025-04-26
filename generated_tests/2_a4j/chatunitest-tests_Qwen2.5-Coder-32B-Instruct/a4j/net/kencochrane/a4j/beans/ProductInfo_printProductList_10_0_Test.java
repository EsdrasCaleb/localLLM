package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductInfo_printProductList_10_0_Test {

    @InjectMocks
    private ProductInfo productInfo;

    @Mock
    private ProductDetails productDetailsMock;

    @BeforeEach
    void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        productDetailsMock = mock(ProductDetails.class);
        when(productDetailsMock.getAsin()).thenReturn("B08N5WRWNW");
        when(productDetailsMock.getProductName()).thenReturn("Sample Product");
        when(productDetailsMock.getOurPrice()).thenReturn("$19.99");
    }

    @Test
    void testPrintProductListWithProducts() throws Exception {
        ArrayList<ProductDetails> products = new ArrayList<>();
        products.add(productDetailsMock);
        Field productsField = ProductInfo.class.getDeclaredField("products");
        productsField.setAccessible(true);
        productsField.set(productInfo, products);
        Field totalResultsField = ProductInfo.class.getDeclaredField("totalResults");
        totalResultsField.setAccessible(true);
        totalResultsField.set(productInfo, "10");
        Field totalPagesField = ProductInfo.class.getDeclaredField("totalPages");
        totalPagesField.setAccessible(true);
        totalPagesField.set(productInfo, "2");
        String expectedOutput = "Total results = 10\n" + "Total pages = 2\n" + "< 0 > B08N5WRWNW : Sample Product - $19.99\n" + "# of products = 1\n";
        assertEquals(expectedOutput, productInfo.printProductList());
    }

    @Test
    void testPrintProductListWithNullProducts() throws Exception {
        Field productsField = ProductInfo.class.getDeclaredField("products");
        productsField.setAccessible(true);
        productsField.set(productInfo, null);
        Field totalResultsField = ProductInfo.class.getDeclaredField("totalResults");
        totalResultsField.setAccessible(true);
        totalResultsField.set(productInfo, "10");
        Field totalPagesField = ProductInfo.class.getDeclaredField("totalPages");
        totalPagesField.setAccessible(true);
        totalPagesField.set(productInfo, "2");
        String expectedOutput = "Total results = 10\n" + "Total pages = 2\n" + "products is null \n";
        assertEquals(expectedOutput, productInfo.printProductList());
    }
}
