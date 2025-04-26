package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Directors_getDirector_3_0_Test {

    @Test
    void getDirector_indexNegative() {
        Directors directors = new Directors();
        String[] directorsArray = { "Director 1", "Director 2" };
        directors.setDirector(directorsArray);
        String director = directors.getDirector(-1);
        assertNull(director);
    }
}
