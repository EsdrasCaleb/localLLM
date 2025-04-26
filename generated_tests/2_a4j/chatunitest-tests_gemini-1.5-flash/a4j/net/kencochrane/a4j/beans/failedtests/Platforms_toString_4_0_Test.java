package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Platforms_toString_4_0_Test {

    @Test
    void testToString_emptyPlatform() {
        Platforms platforms = new Platforms();
        assertEquals("Platforms is null or size 0\n", platforms.toString());
    }

    @Test
    void testToString_nonEmptyPlatform() {
        Platforms platforms = new Platforms();
        String[] platformArray = { "PlatformA", "PlatformB", "PlatformC" };
        platforms.setPlatform(platformArray);
        String expected = "# of Platforms = 3\n" + "Platform - PlatformA\n" + "Platform - PlatformB\n" + "Platform - PlatformC\n";
        assertEquals(expected, platforms.toString());
    }

    @Test
    void testToString_nullPlatform() {
        Platforms platforms = new Platforms();
        try {
            Field platformField = Platforms.class.getDeclaredField("platform");
            platformField.setAccessible(true);
            platformField.set(platforms, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set platform to null: " + e.getMessage());
        }
        assertEquals("Platforms is null or size 0\n", platforms.toString());
    }

    @Test
    void testToString_PlatformWithNullElement() {
        Platforms platforms = new Platforms();
        String[] platformArray = { "PlatformA", null, "PlatformC" };
        platforms.setPlatform(platformArray);
        String expected = "# of Platforms = 3\n" + "Platform - PlatformA\n" + "Platform - null\n" + "Platform - PlatformC\n";
        assertEquals(expected, platforms.toString());
    }

    @Test
    void testToString_PlatformWithEmptyElement() {
        Platforms platforms = new Platforms();
        String[] platformArray = { "PlatformA", "", "PlatformC" };
        platforms.setPlatform(platformArray);
        String expected = "# of Platforms = 3\n" + "Platform - PlatformA\n" + "Platform - \n" + "Platform - PlatformC\n";
        assertEquals(expected, platforms.toString());
    }
}
