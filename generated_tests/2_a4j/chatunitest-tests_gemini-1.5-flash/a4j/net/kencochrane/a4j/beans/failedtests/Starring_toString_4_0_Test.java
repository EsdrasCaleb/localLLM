package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Starring_toString_4_0_Test {

    @Test
    void testToString_WithActors() throws NoSuchFieldException, IllegalAccessException {
        Starring starring = new Starring();
        Field actorsField = Starring.class.getDeclaredField("actors");
        actorsField.setAccessible(true);
        ArrayList<String> actors = new ArrayList<>(Arrays.asList("Actor1", "Actor2", "Actor3"));
        actorsField.set(starring, actors);
        String expected = "# of Actors = 3\n" + "Actor - Actor1\n" + "Actor - Actor2\n" + "Actor - Actor3\n";
        assertEquals(expected, starring.toString());
    }

    @Test
    void testToString_EmptyActors() throws NoSuchFieldException, IllegalAccessException {
        Starring starring = new Starring();
        Field actorsField = Starring.class.getDeclaredField("actors");
        actorsField.setAccessible(true);
        actorsField.set(starring, new ArrayList<>());
        String expected = "Actors is null or size 0\n";
        assertEquals(expected, starring.toString());
    }

    @Test
    void testToString_NullActors() throws NoSuchFieldException, IllegalAccessException {
        Starring starring = new Starring();
        Field actorsField = Starring.class.getDeclaredField("actors");
        actorsField.setAccessible(true);
        actorsField.set(starring, null);
        String expected = "Actors is null or size 0\n";
        assertEquals(expected, starring.toString());
    }

    @Test
    void testToString_ActorWithNullName() throws NoSuchFieldException, IllegalAccessException {
        Starring starring = new Starring();
        Field actorsField = Starring.class.getDeclaredField("actors");
        actorsField.setAccessible(true);
        ArrayList<String> actors = new ArrayList<>(Arrays.asList("Actor1", null, "Actor3"));
        actorsField.set(starring, actors);
        String expected = "# of Actors = 3\n" + "Actor - Actor1\n" + "Actor - null\n" + "Actor - Actor3\n";
        assertEquals(expected, starring.toString());
    }

    @Test
    void testToString_SingleActor() throws NoSuchFieldException, IllegalAccessException {
        Starring starring = new Starring();
        Field actorsField = Starring.class.getDeclaredField("actors");
        actorsField.setAccessible(true);
        ArrayList<String> actors = new ArrayList<>(Arrays.asList("Actor1"));
        actorsField.set(starring, actors);
        String expected = "# of Actors = 1\n" + "Actor - Actor1\n";
        assertEquals(expected, starring.toString());
    }
}
