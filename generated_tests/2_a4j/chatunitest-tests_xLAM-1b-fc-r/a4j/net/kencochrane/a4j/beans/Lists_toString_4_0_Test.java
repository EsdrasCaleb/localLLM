package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Lists_toString_4_0_Test {

    @Test
    public void testToString() throws Exception {
        Lists lists = new Lists();
        Field field = Lists.class.getDeclaredField("lists");
        field.setAccessible(true);
        ArrayList arrayList = new ArrayList();
        arrayList.add("test");
        field.set(lists, arrayList);
        String expected = "# of Lists = 1\n" + "list - test\n";
        assertEquals(expected, lists.toString());
    }
}
