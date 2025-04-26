package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class Starring_getActor_3_1_Test {

    @InjectMocks
    private Starring starring;

    @BeforeEach
    public void setup() {
        starring = new Starring();
        starring.actors = new ArrayList<>();
    }

    @Test
    public void testGetActor_InRange() {
        starring.actors.add("Actor1");
        starring.actors.add("Actor2");
        starring.actors.add("Actor3");
        assertEquals("Actor1", starring.getActor(0));
        assertEquals("Actor2", starring.getActor(1));
        assertEquals("Actor3", starring.getActor(2));
    }

    @Test
    public void testGetActor_OutOfRange() {
        starring.actors.add("Actor1");
        starring.actors.add("Actor2");
        assertNull(starring.getActor(3));
    }

    @Test
    public void testGetActor_NegativeIndex() {
        starring.actors.add("Actor1");
        starring.actors.add("Actor2");
        assertNull(starring.getActor(-1));
    }

    @Test
    public void testGetActor_NullArray() {
        starring.actors = null;
        assertNull(starring.getActor(0));
    }
}
