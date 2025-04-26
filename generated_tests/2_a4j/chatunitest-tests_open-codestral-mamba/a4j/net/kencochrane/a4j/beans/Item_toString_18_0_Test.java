package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Item_toString_18_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Item item;

    @Test
    public void testToString() {
        item.setAsin("AS123456");
        item.setProductName("Test Product");
        item.setQuantity("10");
        String expected = "Asin = AS123456\nName = Test Product\nquantity = 10\n";
        String actual = item.toString();
        assertEquals(expected, actual);
    }
}
