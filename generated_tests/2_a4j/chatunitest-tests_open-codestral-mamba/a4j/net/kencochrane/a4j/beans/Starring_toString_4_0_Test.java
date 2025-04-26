package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Starring_toString_4_0_Test {

    private Starring starring;

    @BeforeEach
    public void setUp() {
        starring = new Starring();
    }

    @Test
    public void testToString_WithActors() {
        ArrayList<String> actors = new ArrayList<>();
        actors.add("Actor 1");
        actors.add("Actor 2");
        starring.setActor(actors.toArray(new String[0]));
        String expected = "# of Actors = 2\n" + "Actor - Actor 1\n" + "Actor - Actor 2\n";
        assertEquals(expected, starring.toString());
    }

    @Test
    public void testToString_WithoutActors() {
        String expected = "Actors is null or size 0\n";
        assertEquals(expected, starring.toString());
    }
}
