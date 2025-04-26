package net.kencochrane.a4j.beans;

import org.junit.Test;
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
        // Arrange
        BrowseList browseList = new BrowseList();
        // Act
        String expected = "# of nodes = 1, BrowseNode[0] = 1, BrowseNode[1] = 2, BrowseNode[2] = 3";
        // Assert
        assertEquals(expected, browseList.toString());
    }
}
