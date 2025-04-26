package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Starring_getActor_3_1_Test {

    @Test
    public void testGetActor() {
        Starring starring = new Starring();
        String[] actors = { "Actor1", "Actor2", "Actor3" };
        starring.setActor(actors);
        String expected = "Actor2";
        String actual = starring.getActor(1);
        assertEquals(expected, actual);
    }
}
