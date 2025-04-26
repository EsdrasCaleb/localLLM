package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class ProductInfo_toString_9_3_Test {

    @Test
    void testToString() {
        ProductInfo productInfo = new ProductInfo();
        productInfo.setTotalResults("10");
        productInfo.setTotalPages("2");
        productInfo.setListName("SampleList");
        productInfo.setDetails(new ProductDetails[] {});
        String expectedOutput = "Total results = 10\nTotal pages = 2\n# of products = 0\n";
        String actualOutput = productInfo.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
