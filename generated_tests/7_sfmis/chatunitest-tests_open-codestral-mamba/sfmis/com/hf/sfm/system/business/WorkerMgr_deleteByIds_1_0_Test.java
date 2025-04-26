package com.hf.sfm.system.business;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.hibernate.Transaction;
import com.hf.sfm.sfmis.personinfo.business.PersonInfoMgr;
import com.hf.sfm.system.pdo.AWorker;
import com.hf.sfm.util.DaoFactory;

public class WorkerMgr_deleteByIds_1_0_Test {

    private WorkerMgr workerMgr;

    @BeforeEach
    public void setUp() {
        workerMgr = Mockito.mock(WorkerMgr.class);
    }

    @Test
    public void testDeleteByIdsSuccess() {
        String[] idnos = { "1", "2", "3" };
        Mockito.when(workerMgr.deleteByIds(idnos)).thenReturn("1");
        String result = workerMgr.deleteByIds(idnos);
        assertEquals("1", result);
    }

    @Test
    public void testDeleteByIdsFailure() {
        String[] idnos = { "1", "2", "3" };
        Mockito.when(workerMgr.deleteByIds(idnos)).thenReturn("0");
        String result = workerMgr.deleteByIds(idnos);
        assertEquals("0", result);
    }
}
