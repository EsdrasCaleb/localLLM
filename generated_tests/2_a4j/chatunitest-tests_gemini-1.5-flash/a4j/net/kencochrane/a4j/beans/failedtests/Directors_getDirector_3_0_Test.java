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

public class Directors_getDirector_3_0_Test {

    private Directors directors;

    @BeforeEach
    void setUp() {
        directors = new Directors();
    }

    @Test
    void testGetDirectorWithinBounds() {
        String[] directorNames = { "Spielberg", "Nolan", "Tarantino" };
        directors.setDirector(directorNames);
        assertEquals("Nolan", directors.getDirector(1));
    }

    @Test
    void testGetDirectorOutOfBounds() {
        String[] directorNames = { "Spielberg", "Nolan", "Tarantino" };
        directors.setDirector(directorNames);
        assertNull(directors.getDirector(3));
    }

    @Test
    void testGetDirectorEmptyList() {
        assertNull(directors.getDirector(0));
    }

    @Test
    void testGetDirectorNegativeIndex() {
        String[] directorNames = { "Spielberg", "Nolan", "Tarantino" };
        directors.setDirector(directorNames);
        assertNull(directors.getDirector(-1));
    }

    @Test
    void testListModification() {
        String[] directorNames = { "Spielberg", "Nolan", "Tarantino" };
        directors.setDirector(directorNames);
        ArrayList<String> internalList = null;
        try {
            Field directorsField = Directors.class.getDeclaredField("directors");
            directorsField.setAccessible(true);
            internalList = (ArrayList<String>) directorsField.get(directors);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access private field 'directors': " + e.getMessage());
        }
        internalList.set(1, "Kubrick");
        assertEquals("Kubrick", directors.getDirector(1));
    }
}
