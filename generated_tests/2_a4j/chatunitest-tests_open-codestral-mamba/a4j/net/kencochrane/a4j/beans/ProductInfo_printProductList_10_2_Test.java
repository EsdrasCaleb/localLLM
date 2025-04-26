package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class ProductInfo_printProductList_10_2_Test {

    @Mock
    private ProductDetails productDetails;

    @InjectMocks
    private ProductInfo productInfo;

    @BeforeEach
    public void setUp() {
        ArrayList<ProductDetails> products = new ArrayList<>();
        products.add(productDetails);
        when(productDetails.getAsin()).thenReturn("ASIN123");
        when(productDetails.getProductName()).thenReturn("Product Name");
        when(productDetails.getOurPrice()).thenReturn("10.00");
        productInfo.setDetails(products.toArray(new ProductDetails[0]));
        productInfo.setTotalResults("10");
        productInfo.setTotalPages("2");
        productInfo.setListName("Test List");
    }

    @Test
    public void testPrintProductList() {
        String expectedOutput = "Total results = 10\n" + "Total pages = 2\n" + "< 0 > ASIN123 : Product Name - 10.00\n" + "# of products = 1\n";
        String actualOutput = productInfo.printProductList();
        assertEquals(expectedOutput, actualOutput);
    }
}
