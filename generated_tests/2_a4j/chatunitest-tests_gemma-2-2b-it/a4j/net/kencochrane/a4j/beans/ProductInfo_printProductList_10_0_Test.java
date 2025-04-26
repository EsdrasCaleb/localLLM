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

    @Test
    void testPrintProductList() {
        ProductInfo productInfo = new ProductInfo();
        productInfo.setTotalResults("10");
        productInfo.setTotalPages("2");
        productInfo.setListName("Test List");
        productInfo.setTotalResults("10");
        productInfo.setTotalPages("2");
        productInfo.setListName("Test List");
        String result = productInfo.printProductList();
        assertEquals("Total results = 10\nTotal pages = 2\n< 0 > 123 : Test Product - 100\n# of products = 1\n", result);
    }
}
