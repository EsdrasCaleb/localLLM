package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Platforms_toString_4_0_Test {

    @Test
    public void testToString_EmptyPlatforms() {
        Platforms platforms = new Platforms();
        String result = platforms.toString();
        assertEquals("Platforms is null or size 0", result);
    }

    @Test
    public void testToString_SinglePlatform() {
        Platforms platforms = new Platforms();
        platforms.setPlatform(new String[] { "Platform1" });
        String result = platforms.toString();
        assertEquals("Platforms is null or size 0", result);
    }

    @Test
    public void testToString_MultiplePlatforms() {
        Platforms platforms = new Platforms();
        platforms.setPlatform(new String[] { "Platform1", "Platform2", "Platform3" });
        String result = platforms.toString();
        assertEquals("Platforms is null or size 0", result);
    }

    @Test
    public void testToString_NullPlatforms() {
        Platforms platforms = new Platforms();
        platforms.setPlatform(null);
        String result = platforms.toString();
        assertEquals("Platforms is null or size 0", result);
    }

    @Test
    public void testToString_EmptyArray() {
        Platforms platforms = new Platforms();
        String[] emptyArray = new String[0];
        platforms.setPlatform(emptyArray);
        String result = platforms.toString();
        assertEquals("Platforms is null or size 0", result);
    }

    @Test
    public void testToString_SingleElementArray() {
        Platforms platforms = new Platforms();
        String[] singleElementArray = new String[] { "Platform1" };
        platforms.setPlatform(singleElementArray);
        String result = platforms.toString();
        assertEquals("Platforms is null or size 0", result);
    }

    @Test
    public void testToString_MultipleElementArray() {
        Platforms platforms = new Platforms();
        String[] multipleElementArray = new String[] { "Platform1", "Platform2", "Platform3" };
        platforms.setPlatform(multipleElementArray);
        String result = platforms.toString();
        assertEquals("Platforms is null or size 0", result);
    }

    @Test
    public void testToString_InvalidInput() {
        Platforms platforms = new Platforms();
        String[] invalidInput = null;
        assertThrows(NullPointerException.class, () -> platforms.setPlatform(invalidInput));
    }
}
