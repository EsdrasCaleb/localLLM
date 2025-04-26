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

    private Platforms platforms;

    @BeforeEach
    public void setUp() {
        platforms = new Platforms();
    }

    @Test
    public void testToStringWithNullPlatform() throws NoSuchFieldException, IllegalAccessException {
        // Set platform to null using reflection
        Field platformField = Platforms.class.getDeclaredField("platform");
        platformField.setAccessible(true);
        platformField.set(platforms, null);
        String expected = "Platforms is null or size 0\n";
        String actual = platforms.toString();
        assertEquals(expected, actual);
    }

    @Test
    public void testToStringWithEmptyPlatform() throws NoSuchFieldException, IllegalAccessException {
        // Set platform to an empty ArrayList using reflection
        Field platformField = Platforms.class.getDeclaredField("platform");
        platformField.setAccessible(true);
        platformField.set(platforms, new ArrayList<>());
        String expected = "Platforms is null or size 0\n";
        String actual = platforms.toString();
        assertEquals(expected, actual);
    }

    @Test
    public void testToStringWithSinglePlatform() throws NoSuchFieldException, IllegalAccessException {
        // Set platform to a list with one element using reflection
        Field platformField = Platforms.class.getDeclaredField("platform");
        platformField.setAccessible(true);
        ArrayList<String> platformsList = new ArrayList<>();
        platformsList.add("Platform1");
        platformField.set(platforms, platformsList);
        String expected = "# of Platforms = 1\nPlatform - Platform1\n";
        String actual = platforms.toString();
        assertEquals(expected, actual);
    }

    @Test
    public void testToStringWithMultiplePlatforms() throws NoSuchFieldException, IllegalAccessException {
        // Set platform to a list with multiple elements using reflection
        Field platformField = Platforms.class.getDeclaredField("platform");
        platformField.setAccessible(true);
        ArrayList<String> platformsList = new ArrayList<>();
        platformsList.add("Platform1");
        platformsList.add("Platform2");
        platformsList.add("Platform3");
        platformField.set(platforms, platformsList);
        String expected = "# of Platforms = 3\nPlatform - Platform1\nPlatform - Platform2\nPlatform - Platform3\n";
        String actual = platforms.toString();
        assertEquals(expected, actual);
    }
}
