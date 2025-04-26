package com.hf.sfm.sfmis.personinfo.business;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import javax.persistence.EntityManager;
import javax.persistence.EntityManagerFactory;
import javax.persistence.Persistence;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.hibernate.Transaction;
import com.hf.sfm.sfmis.personinfo.pdo.APersonInfo;
import com.hf.sfm.util.DaoFactory;

@RunWith(MockitoJUnitRunner.class)
public class PersonInfoMgr_deleteByIds_1_3_Test {

    @Mock
    private EntityManager entityManager;

    @InjectMocks
    private PersonInfoMgr personInfoMgr;

    @Test
    public void testDeleteByIds_Success() {
        // Arrange
        String[] idnos = { "12345", "67890" };
        String expectedRtn = "1";
        // Act
        String actualRtn = personInfoMgr.deleteByIds(idnos);
        // Assert
        assertEquals(expectedRtn, actualRtn);
    }

    @Test
    public void testDeleteByIds_Failure() {
        // Arrange
        String[] idnos = { "12345", "67890" };
        String expectedRtn = "0";
        // Act and Assert
        assertEquals(expectedRtn, personInfoMgr.deleteByIds(idnos));
    }
}
