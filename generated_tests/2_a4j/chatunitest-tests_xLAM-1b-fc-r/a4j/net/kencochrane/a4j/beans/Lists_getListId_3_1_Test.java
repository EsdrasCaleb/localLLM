package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Lists_getListId_3_1_Test {

    @Test
    void getListIdTest() {
        Lists lists = new Lists();
        lists.setListId(new String[] { "1", "2", "3" });
        assertEquals("1", lists.getListId(0));
        assertEquals("2", lists.getListId(1));
        assertEquals("3", lists.getListId(2));
    }
}
