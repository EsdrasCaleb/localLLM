package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class BrowseNode_toString_13_0_Test {

    @Mock
    private BrowseNode browseNode;

    @InjectMocks
    private BrowseNode focalNode;

    @Test
    public void testToString() {
        // Arrange
        when(browseNode.getSubNodes()).thenReturn(new ArrayList<>());
        // Act
        String result = focalNode.toString();
        // Assert
        String expected = "BrowseNode - browseName -- mode";
        assertEquals(expected, result);
    }
}
