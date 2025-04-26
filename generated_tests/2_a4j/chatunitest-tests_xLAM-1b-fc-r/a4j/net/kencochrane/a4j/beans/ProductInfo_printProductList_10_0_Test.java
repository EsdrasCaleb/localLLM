package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class ProductInfo_printProductList_10_0_Test {

    private ProductInfo productInfo;

    @BeforeEach
    public void setUp() {
        productInfo = new ProductInfo();
        productInfo.setListName("testList");
        productInfo.setTotalResults("10");
        productInfo.setTotalPages("5");
        productInfo.setDetails(new ProductDetails[] { new ProductDetails() });
    }

    @Test
    public void testPrintProductList() {
        String expectedOutput = "Total results = 10\n" + "Total pages = 5\n# of products = 1\n< 0 : ProductName - 100\n - ProductDescription\n";
        assertEquals(expectedOutput, productInfo.printProductList());
    }
}
