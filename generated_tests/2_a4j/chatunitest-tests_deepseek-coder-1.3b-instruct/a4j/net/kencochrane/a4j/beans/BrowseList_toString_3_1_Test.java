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
import java.io.Serializable;

public class BrowseList_toString_3_1_Test {

    @Test
    void testToString() throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        BrowseList browseList = new BrowseList();
        BrowseNode browseNode = new BrowseNode();
        browseNode.setBrowseName("TestName");
        browseNode.setBrowseId("TestId");
        ArrayList<BrowseNode> nodes = new ArrayList<>();
        nodes.add(browseNode);
        browseList.setBrowseNode(nodes.toArray(new BrowseNode[0]));
        Method method = BrowseList.class.getDeclaredMethod("toString", null);
        method.setAccessible(true);
        assertEquals("Name: TestName\nID: TestId\n", method.invoke(browseList).toString());
    }
}
