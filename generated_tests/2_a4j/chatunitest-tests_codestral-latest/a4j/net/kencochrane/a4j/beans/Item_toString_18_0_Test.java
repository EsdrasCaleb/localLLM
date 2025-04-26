package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

public class Item_toString_18_0_Test {

    @InjectMocks
    private Item item;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        item.setAsin("B001234567");
        item.setProductName("Sample Product");
        item.setQuantity("10");
    }

    @Test
    public void testToString() {
        String expected = "Asin = B001234567\nName = Sample Product\nquantity = 10\n";
        String result = item.toString();
        assertEquals(expected, result);
    }
}
