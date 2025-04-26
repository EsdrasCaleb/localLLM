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

public class Lists_toString_4_0_Test {

    private Lists listsInstance;

    @BeforeEach
    public void setUp() {
        listsInstance = new Lists();
    }

    @Test
    public void testToString_WhenListsIsNull() throws Exception {
        // Arrange
        Field listsField = Lists.class.getDeclaredField("lists");
        listsField.setAccessible(true);
        listsField.set(listsInstance, null);
        // Act
        String result = listsInstance.toString();
        // Assert
        assertEquals("lists is null or size 0 \n", result);
    }

    @Test
    public void testToString_WhenListsIsEmpty() throws Exception {
        // Arrange
        Field listsField = Lists.class.getDeclaredField("lists");
        listsField.setAccessible(true);
        listsField.set(listsInstance, new ArrayList<>());
        // Act
        String result = listsInstance.toString();
        // Assert
        assertEquals("lists is null or size 0 \n", result);
    }

    @Test
    public void testToString_WhenListsHasElements() throws Exception {
        // Arrange
        ArrayList<String> mockLists = new ArrayList<>();
        mockLists.add("List1");
        mockLists.add("List2");
        Field listsField = Lists.class.getDeclaredField("lists");
        listsField.setAccessible(true);
        listsField.set(listsInstance, mockLists);
        // Act
        String result = listsInstance.toString();
        // Assert
        assertEquals("# of Lists = 2\nlist - List1\nlist - List2\n", result);
    }

    @Test
    public void testToString_WhenListsHasNullElement() throws Exception {
        // Arrange
        ArrayList<String> mockLists = new ArrayList<>();
        mockLists.add(null);
        mockLists.add("List2");
        Field listsField = Lists.class.getDeclaredField("lists");
        listsField.setAccessible(true);
        listsField.set(listsInstance, mockLists);
        // Act
        String result = listsInstance.toString();
        // Assert
        assertEquals("# of Lists = 2\nlist - \nlist - List2\n", result);
    }
}
