package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Starring_toString_4_0_Test {

    private Starring starring;

    @BeforeEach
    void setUp() {
        starring = new Starring();
    }

    @Test
    void testToStringEmpty() {
        assertEquals("Actors is null or size 0\n", starring.toString());
    }

    @Test
    void testToStringNotNull() {
        String[] actors = { "Actor1", "Actor2" };
        starring.setActor(actors);
        String expected = "# of Actors = 2\nActor - Actor1\nActor - Actor2\n";
        assertEquals(expected, starring.toString());
    }

    @Test
    void testToStringNullActors() {
        starring.actors = null;
        assertEquals("Actors is null or size 0\n", starring.toString());
    }

    @Test
    void testToStringWithOneActor() {
        String[] actors = { "Actor1" };
        starring.setActor(actors);
        String expected = "# of Actors = 1\nActor - Actor1\n";
        assertEquals(expected, starring.toString());
    }
}
