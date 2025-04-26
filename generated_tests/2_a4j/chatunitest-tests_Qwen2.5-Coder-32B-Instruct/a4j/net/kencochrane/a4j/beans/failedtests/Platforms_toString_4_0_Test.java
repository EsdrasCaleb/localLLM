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

public class Platforms_toString_4_0_Test {

    @InjectMocks
    private Platforms platforms;

    @Mock
    private ArrayList<String> mockPlatform;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testToString_PlatformIsNull() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        Field platformField = Platforms.class.getDeclaredField("platform");
        platformField.setAccessible(true);
        platformField.set(platforms, null);
        // Act
        String result = platforms.toString();
        // Assert
        assertEquals("Platforms is null or size 0\n", result);
    }

    @Test
    public void testToString_PlatformIsEmpty() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        Field platformField = Platforms.class.getDeclaredField("platform");
        platformField.setAccessible(true);
        platformField.set(platforms, new ArrayList<>());
        // Act
        String result = platforms.toString();
        // Assert
        assertEquals("Platforms is null or size 0\n", result);
    }

    @Test
    public void testToString_PlatformHasElements() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        ArrayList<String> platformList = new ArrayList<>();
        platformList.add("Platform1");
        platformList.add("Platform2");
        Field platformField = Platforms.class.getDeclaredField("platform");
        platformField.setAccessible(true);
        platformField.set(platforms, platformList);
        // Act
        String result = platforms.toString();
        // Assert
        assertEquals("# of Platforms = 2\nPlatform - Platform1\nPlatform - Platform2\n", result);
    }

    @Test
    public void testToString_PlatformHasNullElement() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        ArrayList<String> platformList = new ArrayList<>();
        platformList.add("Platform1");
        platformList.add(null);
        Field platformField = Platforms.class.getDeclaredField("platform");
        platformField.setAccessible(true);
        platformField.set(platforms, platformList);
        // Act
        String result = platforms.toString();
        // Assert
        assertEquals("# of Platforms = 2\nPlatform - Platform1\nPlatform - \n", result);
    }
}
