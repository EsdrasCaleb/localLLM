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

public class Starring_getActor_3_1_Test {

    private Starring starring;

    @BeforeEach
    void setUp() {
        starring = new Starring();
    }

    @Test
    void testGetActorNegativeIndex() {
        String[] actors = { "Actor1", "Actor2", "Actor3" };
        starring.setActor(actors);
        assertNull(starring.getActor(-1));
    }
}
