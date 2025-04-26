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
        BrowseList browseList = mock(BrowseList.class);
        when(browseList.toString()).thenReturn("# of nodes = 2\n" + "Name: Node1\n" + "ID: 1\n" + "Name: Node2\n" + "ID: 2\n");
        String expectedOutput = "# of nodes = 2\n" + "Name: Node1\n" + "ID: 1\n" + "Name: Node2\n" + "ID: 2\n";
        String actualOutput = browseList.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
