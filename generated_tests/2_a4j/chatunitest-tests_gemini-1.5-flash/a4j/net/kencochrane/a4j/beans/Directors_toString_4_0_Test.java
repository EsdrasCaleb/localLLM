package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Directors_toString_4_0_Test {

    @Test
    void testToString_nullDirectors() {
        Directors directors = new Directors();
        assertEquals("Director is null or size 0\n", directors.toString());
    }

    @Test
    void testToString_emptyDirectors() {
        Directors directors = new Directors();
        try {
            Field field = Directors.class.getDeclaredField("directors");
            field.setAccessible(true);
            field.set(directors, new ArrayList<>());
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set directors field: " + e.getMessage());
        }
        assertEquals("Director is null or size 0\n", directors.toString());
    }

    @Test
    void testToString_oneDirector() {
        Directors directors = new Directors();
        try {
            Field field = Directors.class.getDeclaredField("directors");
            field.setAccessible(true);
            field.set(directors, new ArrayList<>(Arrays.asList("Alfred Hitchcock")));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set directors field: " + e.getMessage());
        }
        assertEquals("# of Directors = 1\nDirector - Alfred Hitchcock\n", directors.toString());
    }

    @Test
    void testToString_multipleDirectors() {
        Directors directors = new Directors();
        try {
            Field field = Directors.class.getDeclaredField("directors");
            field.setAccessible(true);
            field.set(directors, new ArrayList<>(Arrays.asList("Alfred Hitchcock", "Steven Spielberg", "Christopher Nolan")));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set directors field: " + e.getMessage());
        }
        String expected = "# of Directors = 3\n" + "Director - Alfred Hitchcock\n" + "Director - Steven Spielberg\n" + "Director - Christopher Nolan\n";
        assertEquals(expected, directors.toString());
    }
}
