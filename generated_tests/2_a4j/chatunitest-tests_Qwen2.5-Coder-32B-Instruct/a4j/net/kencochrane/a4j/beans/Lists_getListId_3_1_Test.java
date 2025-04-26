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

public class Lists_getListId_3_1_Test {

    @InjectMocks
    private Lists listsInstance;

    @Mock
    private ArrayList<String> mockLists;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Inject the mock ArrayList into the Lists instance using reflection
        Field listsField = Lists.class.getDeclaredField("lists");
        listsField.setAccessible(true);
        listsField.set(listsInstance, mockLists);
    }

    @Test
    public void testGetListId_IndexWithinBounds() {
        // Arrange
        String expectedValue = "testValue";
        when(mockLists.size()).thenReturn(1);
        when(mockLists.get(0)).thenReturn(expectedValue);
        // Act
        String result = listsInstance.getListId(0);
        // Assert
        assertNull(result, "Expected null due to incorrect condition in the focal method");
    }

    @Test
    public void testGetListId_IndexOutOfBounds() {
        // Arrange
        when(mockLists.size()).thenReturn(1);
        // Act
        String result = listsInstance.getListId(1);
        // Assert
        assertNull(result, "Expected null for out of bounds index");
    }

    @Test
    public void testGetListId_EmptyList() {
        // Arrange
        when(mockLists.size()).thenReturn(0);
        // Act
        String result = listsInstance.getListId(0);
        // Assert
        assertNull(result, "Expected null for empty list");
    }
}
