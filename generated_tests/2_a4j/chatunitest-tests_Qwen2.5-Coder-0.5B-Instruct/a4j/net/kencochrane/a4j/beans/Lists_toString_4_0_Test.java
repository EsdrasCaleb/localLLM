package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Lists_toString_4_0_Test {

    @Test
    public void testToString() {
        Lists lists = new Lists();
        lists.setListId(new String[] { "list1", "list2", "list3" });
        assertEquals("lists is null or size 0 \nlist - list1\nlist - list2\nlist - list3", lists.toString());
    }
}
