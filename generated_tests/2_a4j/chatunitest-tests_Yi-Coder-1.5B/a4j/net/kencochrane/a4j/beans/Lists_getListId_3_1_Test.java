package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Lists_getListId_3_1_Test {

    @Test
    public void testGetListId() {
        Lists lists = new Lists();
        lists.setListId(new String[] { "1", "2", "3" });
        Assertions.assertArrayEquals(new String[] { "1", "2", "3" }, lists.getListId());
    }
}
