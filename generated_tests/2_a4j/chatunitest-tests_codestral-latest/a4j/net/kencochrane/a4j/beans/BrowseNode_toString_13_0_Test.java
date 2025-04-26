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

class BrowseNode_toString_13_0_Test {

    @InjectMocks
    private BrowseNode browseNode;

    @BeforeEach
    void setUp() throws NoSuchFieldException, IllegalAccessException {
        MockitoAnnotations.openMocks(this);
        // Set private fields using reflection
        Field browseIdField = BrowseNode.class.getDeclaredField("browseId");
        browseIdField.setAccessible(true);
        browseIdField.set(browseNode, "123");
        Field browseNameField = BrowseNode.class.getDeclaredField("browseName");
        browseNameField.setAccessible(true);
        browseNameField.set(browseNode, "Test Node");
        Field modeField = BrowseNode.class.getDeclaredField("mode");
        modeField.setAccessible(true);
        modeField.set(browseNode, "Test Mode");
    }

    @Test
    void testToString() {
        String expected = "123 - Test Node -- Test Mode";
        String result = browseNode.toString();
        assertEquals(expected, result);
    }
}
