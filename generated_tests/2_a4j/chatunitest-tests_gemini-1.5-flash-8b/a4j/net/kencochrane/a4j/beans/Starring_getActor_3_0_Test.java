package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Starring_getActor_3_0_Test {

    @Test
    void testGetActorNegativeIndex() {
        Starring starring = new Starring();
        String[] actorsArray = { "Actor1", "Actor2", "Actor3" };
        starring.setActor(actorsArray);
        String actor = starring.getActor(-1);
        assertNull(actor);
    }
}
