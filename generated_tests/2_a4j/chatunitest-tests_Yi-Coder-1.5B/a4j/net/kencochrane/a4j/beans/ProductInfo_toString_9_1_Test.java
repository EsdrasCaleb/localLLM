package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class ProductInfo_toString_9_1_Test {

    @Test
    void testToString() {
        ProductInfo productInfo = new ProductInfo();
        productInfo.setListName("Name");
        productInfo.setTotalResults("100");
        productInfo.setTotalPages("1");
        productInfo.setDetails(new ProductDetails[0]);
        String expected = "Total results = 100\nTotal pages = 1\n# of products = 0\n";
        String actual = productInfo.toString();
        assertEquals(expected, actual);
    }
}
