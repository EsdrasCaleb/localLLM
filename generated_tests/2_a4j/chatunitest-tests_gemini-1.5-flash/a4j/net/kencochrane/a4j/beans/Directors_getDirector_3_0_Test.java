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
    void testGetDirectorNegativeIndex() {
        String[] directorNames = { "Spielberg", "Nolan", "Tarantino" };
        directors.setDirector(directorNames);
        assertNull(directors.getDirector(-1));
    }
}
