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

public class Platforms_getPlatform_3_0_Test {

    @InjectMocks
    private Platforms platforms;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Initialize the private ArrayList field using reflection
        Field platformField = Platforms.class.getDeclaredField("platform");
        platformField.setAccessible(true);
        platformField.set(platforms, new ArrayList<>());
    }

    @Test
    public void testGetPlatform_withinBounds() throws Exception {
        // Arrange
        ArrayList<String> platformList = new ArrayList<>();
        platformList.add("Platform1");
        platformList.add("Platform2");
        Field platformField = Platforms.class.getDeclaredField("platform");
        platformField.setAccessible(true);
        platformField.set(platforms, platformList);
        // Act
        String result = platforms.getPlatform(1);
        // Assert
        assertEquals("Platform2", result);
    }

    @Test
    public void testGetPlatform_outOfBounds() throws Exception {
        // Arrange
        ArrayList<String> platformList = new ArrayList<>();
        platformList.add("Platform1");
        Field platformField = Platforms.class.getDeclaredField("platform");
        platformField.setAccessible(true);
        platformField.set(platforms, platformList);
        // Act
        String result = platforms.getPlatform(2);
        // Assert
        assertNull(result);
    }

    @Test
    public void testGetPlatform_emptyList() throws Exception {
        ArrayList<String> platformList = new ArrayList<>();
        Field platformField = Platforms.class.getDeclaredField("platform");
        platformField.setAccessible(true);
        platformField.set(platforms, platformList);
        // Act
        String result = platforms.getPlatform(0);
        // Assert
        assertNull(result);
    }

    @Test
    public void testGetPlatform_negativeIndex() throws Exception {
        ArrayList<String> platformList = new ArrayList<>();
        platformList.add("Platform1");
        Field platformField = Platforms.class.getDeclaredField("platform");
        platformField.setAccessible(true);
        platformField.set(platforms, platformList);
        // Act
        String result = platforms.getPlatform(-1);
        // Assert
        assertNull(result);
    }
}
