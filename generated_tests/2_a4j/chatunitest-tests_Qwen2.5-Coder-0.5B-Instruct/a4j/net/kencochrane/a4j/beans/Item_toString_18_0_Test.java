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

class Item_toString_18_0_Test {

    private Item item;

    @BeforeEach
    public void setUp() {
        item = spy(new Item());
    }

    @Test
    public void testToString() {
        String expected = "Asin = asin\nName = productName\nquantity = quantity\n";
        String actual = item.toString();
        assertEquals(expected, actual);
    }
}
