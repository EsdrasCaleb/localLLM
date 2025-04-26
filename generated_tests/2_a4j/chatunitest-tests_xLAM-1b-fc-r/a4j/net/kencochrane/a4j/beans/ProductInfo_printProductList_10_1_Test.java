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

public class ProductInfo_printProductList_10_1_Test {

    private ProductInfo productInfo;

    @BeforeEach
    public void setup() {
        productInfo = new ProductInfo();
        productInfo.setListName("Test List");
        productInfo.setTotalResults("100");
        productInfo.setTotalPages("20");
        ProductDetails product1 = new ProductDetails();
        product1.setAsin("123456");
        product1.setProductName("Product 1");
        product1.setOurPrice("100");
        ProductDetails product2 = new ProductDetails();
        product2.setAsin("654321");
        product2.setProductName("Product 2");
        product2.setOurPrice("200");
        ArrayList<ProductDetails> products = new ArrayList<>();
        products.add(product1);
        products.add(product2);
        productInfo.setDetails(products.toArray(new ProductDetails[0]));
    }

    @Test
    public void testPrintProductList() throws Exception {
        Field field = productInfo.getClass().getDeclaredField("products");
        field.setAccessible(true);
        field.set(productInfo, null);
        String expectedOutput = "Total results = 100\n" + "Total pages = 20\n" + "products is null \n";
        String actualOutput = productInfo.printProductList();
        assertEquals(expectedOutput, actualOutput);
    }
}
