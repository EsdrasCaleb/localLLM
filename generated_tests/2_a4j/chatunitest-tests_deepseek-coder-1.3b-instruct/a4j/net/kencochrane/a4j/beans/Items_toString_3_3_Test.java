package net.kencochrane.a4j.beans;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.Serializable;

public class Items_toString_3_3_Test {

    @Test
    public void testToString() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        Items items = new Items();
        // Testing when the list is empty
        Method method = Items.class.getDeclaredMethod("toString", null);
        String emptyListString = (String) method.invoke(items);
        assertEquals("", emptyListString);
        // Testing when the list contains one item
        Item item = new Item();
        items.getItemsArrayList().add(item);
        method = Items.class.getDeclaredMethod("toString", null);
        String oneItemListString = (String) method.invoke(items);
        assertEquals(item.toString(), oneItemListString);
        // Testing when the list contains multiple items
        Item item2 = new Item();
        items.getItemsArrayList().add(item2);
        method = Items.class.getDeclaredMethod("toString", null);
        String multipleItemsListString = (String) method.invoke(items);
        assertEquals(item.toString() + item2.toString(), multipleItemsListString);
    }
}
