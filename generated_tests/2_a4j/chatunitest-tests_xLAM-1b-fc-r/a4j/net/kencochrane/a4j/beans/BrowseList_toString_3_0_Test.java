package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BrowseList_toString_3_0_Test {

    @Test
    public void testToString() throws NoSuchFieldException, IllegalAccessException {
        BrowseList browseList = new BrowseList();
        Field field = browseList.getClass().getDeclaredField("nodes");
        field.setAccessible(true);
        ArrayList<BrowseNode> nodes = new ArrayList<>();
        nodes.add(new BrowseNode());
        field.set(browseList, nodes);
        String expected = "# of nodes = 1\n" + "Name: \n" + "ID: \n";
        assertEquals(expected, browseList.toString());
    }
}
