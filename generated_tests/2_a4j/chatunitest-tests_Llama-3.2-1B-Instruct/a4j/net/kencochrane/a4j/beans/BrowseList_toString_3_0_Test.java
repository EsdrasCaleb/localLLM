package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class BrowseList_toString_3_0_Test {

    @Test
    public void testToString() {
        BrowseList browseList = new BrowseList();
        String expected = "# of nodes = 0\n";
        expected += "No nodes\n";
        assertEquals(expected, browseList.toString());
    }
}
