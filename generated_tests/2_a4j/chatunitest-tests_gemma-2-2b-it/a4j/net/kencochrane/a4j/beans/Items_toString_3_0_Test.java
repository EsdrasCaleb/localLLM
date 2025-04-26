package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.Serializable;
import java.util.ArrayList;

public class Items_toString_3_0_Test {

    @Test
    void testToString() {
        Items items = new Items();
        items.setItem(new Item[] { new Item(), new Item() });
        String result = items.toString();
        assertEquals("Item1,Item2", result);
    }
}
