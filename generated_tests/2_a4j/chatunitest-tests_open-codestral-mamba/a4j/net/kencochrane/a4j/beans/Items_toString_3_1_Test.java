package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Items_toString_3_1_Test {

    @Mock
    private Item item;

    @InjectMocks
    private Items items;

    @BeforeEach
    public void setUp() {
        ArrayList<Item> itemList = new ArrayList<>();
        itemList.add(item);
        when(item.toString()).thenReturn("Item1");
        items.setItem(itemList.toArray(new Item[0]));
    }

    @Test
    public void testToString() {
        String expected = "Item1";
        String actual = items.toString();
        assertEquals(expected, actual);
    }
}
